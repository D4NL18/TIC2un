import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SvmResultsComponent } from './svm-results.component';

describe('SvmResultsComponent', () => {
  let component: SvmResultsComponent;
  let fixture: ComponentFixture<SvmResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SvmResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SvmResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
