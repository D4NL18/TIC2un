import { TestBed } from '@angular/core/testing';

import { SomService } from './som.service';

describe('SomService', () => {
  let service: SomService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SomService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
