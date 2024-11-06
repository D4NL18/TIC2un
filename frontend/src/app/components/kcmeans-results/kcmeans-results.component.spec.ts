import { ComponentFixture, TestBed } from '@angular/core/testing';

import { KcmeansResultsComponent } from './kcmeans-results.component';

describe('KcmeansResultsComponent', () => {
  let component: KcmeansResultsComponent;
  let fixture: ComponentFixture<KcmeansResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [KcmeansResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(KcmeansResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
