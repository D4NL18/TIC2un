import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CnnResultsComponent } from './cnn-results.component';

describe('CnnResultsComponent', () => {
  let component: CnnResultsComponent;
  let fixture: ComponentFixture<CnnResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CnnResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CnnResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
